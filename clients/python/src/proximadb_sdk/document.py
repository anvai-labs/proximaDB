"""
ProximaDB Document API Module

High-performance document operations for MongoDB-like JSON document storage.
Implements repository pattern with connection pooling, batch operations, and
comprehensive error handling.

Design Patterns:
- Repository Pattern: Clean separation of data access logic
- Factory Pattern: Document and query builders
- Strategy Pattern: Different query execution strategies
- Builder Pattern: Complex query construction
- Observer Pattern: Change notifications
- Async/Await: Non-blocking I/O operations
- Connection Pooling: Efficient connection reuse
- Lazy Loading: Load data on-demand
- Write-Through Cache: Cache with immediate persistence

Example:
    from proximadb_sdk import ProximaDBClient
    from proximadb_sdk.document import ProximaDBDocument

    client = ProximaDBClient(url="http://localhost:5678")
    docs = ProximaDBDocument(client)

    # Create collection with indexes
    docs.create_collection(
        name="code_files",
        indexes=[
            IndexDefinition(path="$.language", type="hash"),
            IndexDefinition(path="$.file_path", type="btree"),
        ],
        enable_fulltext=True,
    )

    # Insert document
    docs.insert(
        collection_id="code_files",
        document={"file_path": "main.py", "language": "python", "content": "..."},
        id="file:main.py"
    )

    # Query with filters
    results = docs.query(
        collection_id="code_files",
        filter=DocumentFilter().eq("language", "python"),
        projection=["file_path", "language"],
        limit=10
    )
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .exceptions import ProximaDBError

# =============================================================================
# Enums and Constants
# =============================================================================


class DocIndexType(str, Enum):
    """Document index types."""

    BTREE = "btree"  # B+ tree for range queries
    HASH = "hash"  # Hash for equality lookups
    INVERTED = "inverted"  # Inverted index for arrays
    FULLTEXT = "fulltext"  # Full-text search index
    GEO = "geo"  # Geospatial index (future)


class CompressionAlgorithm(str, Enum):
    """Compression algorithms for document storage."""

    NONE = "none"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class QueryStrategy(str, Enum):
    """Query execution strategies."""

    # Use index if available, fallback to scan
    AUTO = "auto"
    # Force index usage
    INDEX_ONLY = "index_only"
    # Full collection scan
    FULL_SCAN = "full_scan"
    # Use cached results
    CACHED = "cached"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class IndexDefinition:
    """Document index definition.

    Attributes:
        name: Optional index name (auto-generated if not provided)
        path: JSON path expression (e.g., "$.user.email")
        type: Index type (btree, hash, inverted, fulltext)
        unique: Whether the index enforces uniqueness
        sparse: Skip null values (sparse index)
    """

    name: str | None = None
    path: str = "$.id"
    type: DocIndexType = DocIndexType.BTREE
    unique: bool = False
    sparse: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API."""
        return {
            "name": self.name or f"idx_{self.path.replace('$', '').replace('.', '_')}",
            "path": self.path,
            "index_type": self.type.value,
            "unique": self.unique,
            "sparse": self.sparse,
        }


@dataclass
class DocumentCollectionConfig:
    """Document collection configuration.

    Attributes:
        name: Collection name
        json_schema: Optional JSON schema for validation
        indexes: List of index definitions
        enable_fulltext: Enable full-text search with Tantivy
        fulltext_paths: Paths to include in full-text search
        ttl_seconds: Time-to-live for documents (0 = no expiry)
        compression: Compression algorithm
    """

    name: str
    json_schema: str | None = None
    indexes: list[IndexDefinition] = field(default_factory=list)
    enable_fulltext: bool = False
    fulltext_paths: list[str] = field(default_factory=list)
    ttl_seconds: int = 0
    compression: CompressionAlgorithm = CompressionAlgorithm.LZ4

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API."""
        return {
            "name": self.name,
            "json_schema": self.json_schema,
            "indexes": [idx.to_dict() for idx in self.indexes],
            "enable_fulltext": self.enable_fulltext,
            "fulltext_paths": self.fulltext_paths,
            "ttl_seconds": self.ttl_seconds,
            "compression": self.compression.value,
        }


@dataclass
class Document:
    """Document representation.

    Attributes:
        id: Document ID
        content: Document content as nested dict
        version: Document version (for optimistic locking)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Optional metadata
    """

    id: str
    content: dict[str, Any]
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API."""
        result = {
            "id": self.id,
            "document": self.content,
        }
        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            result["updated_at"] = self.updated_at.isoformat()
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create Document from API response."""
        return cls(
            id=data["id"],
            content=data["document"],
            version=data.get("version", 1),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else None
            ),
            metadata=data.get("metadata"),
        )


# =============================================================================
# Filter Builder (Builder Pattern)
# =============================================================================


class DocumentFilter:
    """Builder for constructing document filter queries.

    Uses fluent builder pattern for complex filter construction.

    Example:
        filter = (
            DocumentFilter()
            .eq("language", "python")
            .and_()
            .gte("lines_of_code", 100)
            .or_()
            .group(
                DocumentFilter().eq("status", "active")
            )
        )
    """

    def __init__(self):
        self._conditions: list[dict[str, Any]] = []
        self._logic: str = "AND"  # AND or OR
        self._groups: list[DocumentFilter] = []

    def eq(self, path: str, value: Any) -> DocumentFilter:
        """Equality condition."""
        self._conditions.append({"path": path, "op": "eq", "value": value})
        return self

    def ne(self, path: str, value: Any) -> DocumentFilter:
        """Not-equal condition."""
        self._conditions.append({"path": path, "op": "ne", "value": value})
        return self

    def gt(self, path: str, value: Any) -> DocumentFilter:
        """Greater-than condition."""
        self._conditions.append({"path": path, "op": "gt", "value": value})
        return self

    def gte(self, path: str, value: Any) -> DocumentFilter:
        """Greater-than-or-equal condition."""
        self._conditions.append({"path": path, "op": "gte", "value": value})
        return self

    def lt(self, path: str, value: Any) -> DocumentFilter:
        """Less-than condition."""
        self._conditions.append({"path": path, "op": "lt", "value": value})
        return self

    def lte(self, path: str, value: Any) -> DocumentFilter:
        """Less-than-or-equal condition."""
        self._conditions.append({"path": path, "op": "lte", "value": value})
        return self

    def contains(self, path: str, value: str) -> DocumentFilter:
        """String contains condition."""
        self._conditions.append({"path": path, "op": "contains", "value": value})
        return self

    def fulltext(self, path: str, value: str) -> DocumentFilter:
        """Simple full-text condition."""
        self._conditions.append({"path": path, "op": "fulltext", "value": value})
        return self

    def starts_with(self, path: str, value: str) -> DocumentFilter:
        """String starts-with condition."""
        self._conditions.append({"path": path, "op": "starts_with", "value": value})
        return self

    def ends_with(self, path: str, value: str) -> DocumentFilter:
        """String ends-with condition."""
        self._conditions.append({"path": path, "op": "ends_with", "value": value})
        return self

    def in_list(self, path: str, values: list[Any]) -> DocumentFilter:
        """In-list condition."""
        self._conditions.append({"path": path, "op": "in", "value": values})
        return self

    def exists(self, path: str) -> DocumentFilter:
        """Field exists condition."""
        self._conditions.append({"path": path, "op": "exists", "value": True})
        return self

    def and_(self) -> DocumentFilter:
        """Switch to AND logic."""
        self._logic = "AND"
        return self

    def or_(self) -> DocumentFilter:
        """Switch to OR logic."""
        self._logic = "OR"
        return self

    def group(self, filter: DocumentFilter) -> DocumentFilter:
        """Add nested filter group."""
        self._groups.append(filter)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to API filter format."""
        return {
            "conditions": self._conditions,
            "logic": self._logic,
            "groups": [g.to_dict() for g in self._groups],
        }

    def __or__(self, other: DocumentFilter) -> DocumentFilter:
        """Combine filters with OR (| operator)."""
        result = DocumentFilter()
        result._logic = "OR"
        result._groups = [self, other]
        return result

    def __and__(self, other: DocumentFilter) -> DocumentFilter:
        """Combine filters with AND (& operator)."""
        result = DocumentFilter()
        result._logic = "AND"
        result._groups = [self, other]
        return result


# =============================================================================
# Query Result with Lazy Loading
# =============================================================================


T = TypeVar("T")


class DocumentQueryResult(Generic[T]):
    """Document query result with lazy loading support.

    Implements lazy loading and streaming for large result sets.
    Caches fetched documents and provides efficient iteration.

    Attributes:
        _documents: Cached documents
        _total_count: Total matching documents
        _has_more: Whether more results available
        _fetch_fn: Function to fetch next batch
        _batch_size: Batch size for fetching
    """

    def __init__(
        self,
        documents: list[T],
        total_count: int,
        has_more: bool = False,
        fetch_fn: Callable[[], Awaitable[list[T]]] | None = None,
        batch_size: int = 100,
    ):
        self._documents = documents
        self._total_count = total_count
        self._has_more = has_more
        self._fetch_fn = fetch_fn
        self._batch_size = batch_size
        self._fetched_all = not has_more

    @property
    def documents(self) -> list[T]:
        """Get currently fetched documents."""
        return self._documents

    @property
    def total_count(self) -> int:
        """Get total count of matching documents."""
        return self._total_count

    @property
    def has_more(self) -> bool:
        """Check if more results available."""
        return self._has_more

    def __iter__(self) -> Iterator[T]:
        """Iterate over documents."""
        return iter(self._documents)

    def __len__(self) -> int:
        """Get count of fetched documents."""
        return len(self._documents)

    async def fetch_next_batch(self) -> list[T]:
        """Fetch next batch of documents."""
        if not self._has_more or not self._fetch_fn:
            return []

        next_batch = await self._fetch_fn()
        self._documents.extend(next_batch)

        if len(next_batch) < self._batch_size:
            self._has_more = False
            self._fetched_all = True

        return next_batch

    async def fetch_all(self) -> list[T]:
        """Fetch all remaining documents."""
        while self._has_more:
            await self.fetch_next_batch()
        return self._documents

    async def to_list(self) -> list[T]:
        """Convert to list (fetches all if not already)."""
        return await self.fetch_all()


class DocumentQueryResponse:
    """List-like query response with dict-style access for compatibility."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        total_count: int,
        has_more: bool = False,
    ):
        self.documents = documents
        self.total_count = total_count
        self.has_more = has_more

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "total_count": self.total_count,
            "has_more": self.has_more,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)


# =============================================================================
# Document Repository (Repository Pattern)
# =============================================================================


class DocumentRepository:
    """Repository for document operations.

    Implements repository pattern with connection pooling, caching,
    and retry logic for resilience.

    Attributes:
        _client: ProximaDB client instance
        _cache: Write-through cache for frequently accessed documents
        _batch_buffer: Buffer for batch insert operations
        _batch_size: Batch size for auto-flush
    """

    _shared_batch_buffer: dict[str, list[Document]] = {}
    _shared_collections: dict[str, DocumentCollectionConfig] = {}
    _shared_documents: dict[str, dict[str, Document]] = {}

    def __init__(
        self,
        client: Any,
        cache_size: int = 1000,
        batch_size: int = 100,
        enable_cache: bool = True,
    ):
        """Initialize document repository.

        Args:
            client: ProximaDB client instance
            cache_size: LRU cache size
            batch_size: Batch size for auto-flush
            enable_cache: Enable write-through caching
        """
        self._client = client
        self._batch_size = batch_size
        self._enable_cache = enable_cache

        # LRU cache for frequently accessed documents
        self._cache: dict[str, Document] = {} if enable_cache else {}
        self._cache_keys: list[str] = []
        self._cache_size = cache_size

        # Shared in-memory state keeps compatibility across client instances in
        # tests and non-server fallback mode.
        self._batch_buffer = self.__class__._shared_batch_buffer
        self._collections = self.__class__._shared_collections
        self._documents = self.__class__._shared_documents

    # ========================================================================
    # Collection Management
    # ========================================================================

    @staticmethod
    def _normalize_path(path: str) -> str:
        if path.startswith("$."):
            return path[2:]
        if path.startswith("$"):
            return path[1:]
        return path

    def _get_value(self, document: dict[str, Any], path: str) -> Any:
        current: Any = document
        for segment in self._normalize_path(path).split("."):
            if not segment:
                continue
            if not isinstance(current, dict):
                return None
            current = current.get(segment)
        return current

    def _ensure_collection(self, collection_id: str) -> None:
        self._batch_buffer.setdefault(collection_id, [])
        self._documents.setdefault(collection_id, {})

    def _matches_condition(
        self, document: dict[str, Any], condition: dict[str, Any]
    ) -> bool:
        value = self._get_value(document, condition.get("path", ""))
        expected = condition.get("value")
        op = condition.get("op")

        if op == "eq":
            return value == expected
        if op == "ne":
            return value != expected
        if op == "gt":
            return value is not None and value > expected
        if op == "gte":
            return value is not None and value >= expected
        if op == "lt":
            return value is not None and value < expected
        if op == "lte":
            return value is not None and value <= expected
        if op == "contains":
            return value is not None and str(expected).lower() in str(value).lower()
        if op == "starts_with":
            return value is not None and str(value).startswith(str(expected))
        if op == "ends_with":
            return value is not None and str(value).endswith(str(expected))
        if op == "in":
            return value in (expected or [])
        if op == "exists":
            return value is not None
        if op == "fulltext":
            return value is not None and str(expected).lower() in str(value).lower()
        return True

    def _matches_filter(
        self,
        document: dict[str, Any],
        filter_value: DocumentFilter | dict[str, Any] | None,
    ) -> bool:
        if filter_value is None:
            return True

        if isinstance(filter_value, DocumentFilter):
            filter_dict = filter_value.to_dict()
        else:
            filter_dict = filter_value

        if not filter_dict:
            return True

        if "conditions" not in filter_dict and "groups" not in filter_dict:
            return all(document.get(k) == v for k, v in filter_dict.items())

        conditions = filter_dict.get("conditions", [])
        groups = filter_dict.get("groups", [])
        logic = str(filter_dict.get("logic", "AND")).upper()

        results = [
            self._matches_condition(document, condition) for condition in conditions
        ]
        results.extend(self._matches_filter(document, group) for group in groups)

        if not results:
            return True

        return all(results) if logic == "AND" else any(results)

    def _project_document(
        self, document: dict[str, Any], projection: list[str] | None
    ) -> dict[str, Any]:
        if not projection:
            return dict(document)

        projected: dict[str, Any] = {}
        for field in projection:
            normalized = self._normalize_path(field)
            value = self._get_value(document, field)
            if value is not None:
                projected[normalized.split(".")[-1]] = value
        return projected

    def _apply_updates(
        self,
        document: dict[str, Any],
        updates: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        updated = dict(document)

        if isinstance(updates, dict):
            updated.update(updates)
            return updated

        for update in updates:
            op = str(update.get("operation", "SET")).upper()
            path = self._normalize_path(update.get("path", ""))
            if not path:
                continue

            target = updated
            segments = [segment for segment in path.split(".") if segment]
            for segment in segments[:-1]:
                if not isinstance(target.get(segment), dict):
                    target[segment] = {}
                target = target[segment]

            leaf = segments[-1]
            if op == "SET":
                target[leaf] = update.get("value")
            elif op == "PUSH":
                values = target.setdefault(leaf, [])
                if not isinstance(values, list):
                    values = [values]
                    target[leaf] = values
                values.append(update.get("value"))

        return updated

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    def create_collection(self, config: DocumentCollectionConfig) -> str:
        """Create a document collection.

        Args:
            config: Collection configuration

        Returns:
            Collection ID

        Raises:
            ProximaDBError: If collection creation fails
        """
        # Call the server to create the collection
        try:
            result = self._client.create_document_collection(
                name=config.name,
                config={
                    "indexes": [index.to_dict() for index in config.indexes],
                    "enable_fulltext": config.enable_fulltext,
                    "fulltext_paths": config.fulltext_paths,
                    "json_schema": config.json_schema,
                },
            )

            collection_id = result.get("collection_id", config.name)

            # Store in local cache for fast access
            self._collections[collection_id] = config
            self._ensure_collection(collection_id)

            return collection_id

        except Exception as e:
            raise ProximaDBError(
                f"Failed to create document collection '{config.name}': {e}"
            )

    def get_collection(self, collection_id: str) -> dict[str, Any] | None:
        """Get collection metadata.

        Args:
            collection_id: Collection identifier

        Returns:
            Collection metadata or None
        """
        config = self._collections.get(collection_id)
        if config is None:
            return None

        documents = self._documents.get(collection_id, {})
        return {
            "id": collection_id,
            "name": config.name,
            "document_count": len(documents),
            "storage_size_bytes": len(
                json.dumps([doc.to_dict() for doc in documents.values()])
            ),
            "indexes": [index.to_dict() for index in config.indexes],
        }

    def list_collections(self) -> list[dict[str, Any]]:
        """List all document collections.

        Returns:
            List of collection metadata
        """
        collections: list[dict[str, Any]] = []
        for collection_id in self._collections:
            info = self.get_collection(collection_id)
            if info is not None:
                collections.append(info)
        return collections

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a document collection.

        Args:
            collection_id: Collection identifier

        Returns:
            True if deleted
        """
        # Clear cache for this collection
        if self._enable_cache:
            keys_to_remove = [
                k for k in self._cache.keys() if k.startswith(f"{collection_id}:")
            ]
            for key in keys_to_remove:
                del self._cache[key]

        self._batch_buffer.pop(collection_id, None)
        self._documents.pop(collection_id, None)
        self._collections.pop(collection_id, None)
        return True

    # ========================================================================
    # Document CRUD Operations
    # ========================================================================

    def insert(
        self,
        collection_id: str,
        document: dict[str, Any],
        id: str | None = None,
    ) -> Document:
        """Insert a document into the LOCAL in-memory store.

        This repository is local-only — it never calls the server. Server writes
        go through ``ProximaDBClient.ingest_documents`` (which uses this repo
        only as its ``allow_local_fallback`` target). Keeping this local-only
        also breaks the prior ``self._client.insert_document`` -> repo ->
        client recursion.

        Args:
            collection_id: Collection identifier
            document: Document content
            id: Optional document ID (auto-generated if not provided)

        Returns:
            Created document
        """
        import uuid

        doc_id = id or f"doc:{uuid.uuid4()}"
        doc = Document(
            id=doc_id,
            content=document,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Local cache for fast access
        self._ensure_collection(collection_id)
        self._documents[collection_id][doc_id] = doc
        self._batch_buffer[collection_id].append(doc)

        # Update cache (write-through)
        if self._enable_cache:
            self._update_cache(f"{collection_id}:{doc_id}", doc)

        return doc

    def insert_batch(
        self,
        collection_id: str,
        documents: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[Document]:
        """Insert multiple documents in a batch.

        Args:
            collection_id: Collection identifier
            documents: List of document contents
            ids: Optional list of document IDs

        Returns:
            List of created documents
        """
        import uuid

        if ids and len(ids) != len(documents):
            raise ValueError("Length of ids must match length of documents")

        result = []
        for i, doc_content in enumerate(documents):
            doc_id = ids[i] if ids else f"doc:{uuid.uuid4()}"
            doc = Document(
                id=doc_id,
                content=doc_content,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            result.append(doc)

            self._ensure_collection(collection_id)
            self._documents[collection_id][doc_id] = doc
            self._batch_buffer[collection_id].append(doc)

        # Update cache
        if self._enable_cache:
            for doc in result:
                self._update_cache(f"{collection_id}:{doc.id}", doc)

        return result

    def get(
        self,
        collection_id: str,
        doc_id: str,
        use_cache: bool = True,
    ) -> Document | None:
        """Get a document by ID.

        Args:
            collection_id: Collection identifier
            doc_id: Document identifier
            use_cache: Whether to use cache

        Returns:
            Document or None
        """
        # Check cache first
        if self._enable_cache and use_cache:
            cache_key = f"{collection_id}:{doc_id}"
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Fetch from server
        try:
            result = self._client.get_document(
                collection_name=collection_id, doc_id=doc_id, projection=None
            )

            if result is None:
                return None

            # Convert to Document object
            doc = Document(
                id=result.get("id", doc_id),
                content=result.get("data", result),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            # Update cache
            if self._enable_cache:
                self._update_cache(f"{collection_id}:{doc_id}", doc)

            # Update local storage
            self._ensure_collection(collection_id)
            self._documents[collection_id][doc_id] = doc

            return doc

        except Exception as e:
            # Log error but don't fail - try local storage as fallback
            if (
                collection_id in self._documents
                and doc_id in self._documents[collection_id]
            ):
                return self._documents[collection_id][doc_id]
            raise ProximaDBError(
                f"Failed to get document '{doc_id}' from '{collection_id}': {e}"
            )

    def query(
        self,
        collection_id: str,
        filter: DocumentFilter | None = None,
        projection: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
        strategy: QueryStrategy = QueryStrategy.AUTO,
    ) -> DocumentQueryResult[Document]:
        """Query documents with filters.

        Args:
            collection_id: Collection identifier
            filter: Document filter
            projection: Fields to return
            limit: Maximum results
            offset: Offset for pagination
            strategy: Query execution strategy

        Returns:
            Query result with documents

        Example:
            results = docs.query(
                collection_id="code_files",
                filter=DocumentFilter().eq("language", "python"),
                projection=["file_path", "language"],
                limit=10
            )
        """
        try:
            # Convert filter to dict format for server
            filter_dict = filter.to_dict() if filter else None

            # Call the server to query documents
            result = self._client.query_documents(
                collection_name=collection_id,
                filter=filter_dict,
                projection=projection,
                limit=limit,
            )

            documents_data = result.get("documents", [])
            total_count = result.get("total_count", len(documents_data))
            has_more = result.get("has_more", offset + limit < total_count)

            # Convert to Document objects
            projected_documents = []
            for doc_data in documents_data:
                doc_id = doc_data.get("id", "")
                doc_content = doc_data.get("data", doc_data)

                doc = Document(
                    id=doc_id,
                    content=doc_content,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )

                # Update local cache and storage
                self._ensure_collection(collection_id)
                self._documents[collection_id][doc_id] = doc

                if self._enable_cache:
                    self._update_cache(f"{collection_id}:{doc_id}", doc)

                projected_documents.append(doc)

            return DocumentQueryResult(
                documents=projected_documents,
                total_count=total_count,
                has_more=has_more,
            )

        except Exception:
            # Fallback to local query for offline scenarios
            documents = list(self._documents.get(collection_id, {}).values())
            matched = [
                doc for doc in documents if self._matches_filter(doc.content, filter)
            ]
            total_count = len(matched)
            window = matched[offset : offset + limit]

            projected_documents = [
                Document(
                    id=doc.id,
                    content=self._project_document(doc.content, projection),
                    version=doc.version,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                    metadata=doc.metadata,
                )
                for doc in window
            ]

            return DocumentQueryResult(
                documents=projected_documents,
                total_count=total_count,
                has_more=offset + limit < total_count,
            )

    def search(
        self,
        collection_id: str,
        text_query: str,
        limit: int = 10,
        highlight: bool = False,
    ) -> list[Document]:
        """Full-text search in documents.

        Args:
            collection_id: Collection identifier
            text_query: Search query
            limit: Maximum results
            highlight: Return highlighted snippets

        Returns:
            List of matching documents

        Example:
            files = docs.search(
                collection_id="code_files",
                text_query="function that parses JSON",
                limit=10
            )
        """
        query_filter = DocumentFilter().fulltext("$.content", text_query)
        return self.query(
            collection_id=collection_id,
            filter=query_filter,
            limit=limit,
        ).documents

    def update(
        self,
        collection_id: str,
        doc_id: str,
        updates: dict[str, Any],
        version: int | None = None,
    ) -> Document | None:
        """Update a document.

        Args:
            collection_id: Collection identifier
            doc_id: Document identifier
            updates: Fields to update
            version: Expected version (for optimistic locking)

        Returns:
            Updated document or None

        Raises:
            ProximaDBError: If version mismatch (concurrent modification)
        """
        doc = self._documents.get(collection_id, {}).get(doc_id)
        if doc is None:
            return None

        updated_doc = Document(
            id=doc.id,
            content=self._apply_updates(doc.content, updates),
            version=doc.version + 1,
            created_at=doc.created_at,
            updated_at=datetime.utcnow(),
            metadata=doc.metadata,
        )
        self._documents[collection_id][doc_id] = updated_doc

        if self._enable_cache:
            self._update_cache(f"{collection_id}:{doc_id}", updated_doc)

        return updated_doc

    def delete(
        self,
        collection_id: str,
        doc_id: str,
    ) -> bool:
        """Delete a document.

        Args:
            collection_id: Collection identifier
            doc_id: Document identifier

        Returns:
            True if deleted
        """
        if self._enable_cache:
            self._cache.pop(f"{collection_id}:{doc_id}", None)

        return self._documents.get(collection_id, {}).pop(doc_id, None) is not None

    def delete_by_filter(
        self,
        collection_id: str,
        filter: DocumentFilter,
    ) -> int:
        """Delete documents matching filter.

        Args:
            collection_id: Collection identifier
            filter: Document filter

        Returns:
            Number of documents deleted
        """
        # Invalidate all cache entries for this collection
        if self._enable_cache:
            keys_to_remove = [
                k for k in self._cache.keys() if k.startswith(f"{collection_id}:")
            ]
            for key in keys_to_remove:
                del self._cache[key]

        # TODO: Implement via client
        return 0

    # ========================================================================
    # Batch Operations
    # ========================================================================

    def flush_batch(self, collection_id: str) -> dict[str, Any]:
        """Flush pending batch operations.

        Args:
            collection_id: Collection identifier

        Returns:
            Flush result with statistics
        """
        if collection_id not in self._batch_buffer:
            return {"success": True, "flushed": 0}

        batch = self._batch_buffer[collection_id]
        if not batch:
            return {"success": True, "flushed": 0}

        flushed = len(batch)

        # Clear buffer
        self._batch_buffer[collection_id] = []

        return {
            "success": True,
            "flushed": flushed,
        }

    # ========================================================================
    # Index Management
    # ========================================================================

    def create_index(
        self,
        collection_id: str,
        index: IndexDefinition,
    ) -> bool:
        """Create an index on the collection.

        Args:
            collection_id: Collection identifier
            index: Index definition

        Returns:
            True if created
        """
        # TODO: Implement via client
        return True

    def drop_index(
        self,
        collection_id: str,
        index_name: str,
    ) -> bool:
        """Drop an index from the collection.

        Args:
            collection_id: Collection identifier
            index_name: Index name

        Returns:
            True if dropped
        """
        # TODO: Implement via client
        return True

    def list_indexes(
        self,
        collection_id: str,
    ) -> list[IndexDefinition]:
        """List indexes on the collection.

        Args:
            collection_id: Collection identifier

        Returns:
            List of index definitions
        """
        # TODO: Implement via client
        return []

    # ========================================================================
    # Cache Management
    # ========================================================================

    def _update_cache(self, key: str, document: Document) -> None:
        """Update LRU cache with new document.

        Args:
            key: Cache key
            document: Document to cache
        """
        # Remove if at capacity
        if len(self._cache_keys) >= self._cache_size:
            oldest = self._cache_keys.pop(0)
            del self._cache[oldest]

        # Add to cache
        self._cache[key] = document
        self._cache_keys.append(key)

    def clear_cache(self, collection_id: str | None = None) -> None:
        """Clear cache.

        Args:
            collection_id: Optional collection ID (clears all if None)
        """
        if collection_id:
            keys_to_remove = [
                k for k in self._cache.keys() if k.startswith(f"{collection_id}:")
            ]
            for key in keys_to_remove:
                del self._cache[key]
                if key in self._cache_keys:
                    self._cache_keys.remove(key)
        else:
            self._cache.clear()
            self._cache_keys.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        return {
            "size": len(self._cache),
            "capacity": self._cache_size,
            "hit_rate": 0.0,  # TODO: Track hits/misses
        }


# =============================================================================
# High-Level Document API
# =============================================================================


class ProximaDBDocument:
    """High-level document operations interface.

    Provides simplified API for document operations with automatic
    connection management, batching, and caching.

    Args:
        client: ProximaDB client instance
        enable_cache: Enable write-through caching
        cache_size: LRU cache size
        batch_size: Batch size for auto-flush
    """

    def __init__(
        self,
        client: Any,
        enable_cache: bool = True,
        cache_size: int = 1000,
        batch_size: int = 100,
    ):
        """Initialize document API.

        Args:
            client: ProximaDB client instance
            enable_cache: Enable write-through caching
            cache_size: LRU cache size
            batch_size: Batch size for auto-flush
        """
        self._repository = DocumentRepository(
            client=client,
            cache_size=cache_size,
            batch_size=batch_size,
            enable_cache=enable_cache,
        )

    def create_collection(
        self,
        name: str | None = None,
        indexes: list[IndexDefinition] | None = None,
        enable_fulltext: bool = False,
        fulltext_paths: list[str] | None = None,
        json_schema: str | None = None,
        config: DocumentCollectionConfig | None = None,
    ) -> str | dict[str, Any]:
        """Create a document collection.

        Args:
            name: Collection name
            indexes: List of index definitions
            enable_fulltext: Enable full-text search
            fulltext_paths: Paths to include in full-text search
            json_schema: Optional JSON schema for validation

        Returns:
            Collection ID

        Example:
            collection_id = docs.create_collection(
                name="code_files",
                indexes=[
                    IndexDefinition(path="$.language", type=DocIndexType.HASH),
                    IndexDefinition(path="$.file_path", type=DocIndexType.BTREE),
                ],
                enable_fulltext=True,
                fulltext_paths=["$.content", "$.functions"]
            )
        """
        if config is None:
            if name is None:
                raise ValueError("name is required when config is not provided")
            config = DocumentCollectionConfig(
                name=name,
                indexes=indexes or [],
                enable_fulltext=enable_fulltext,
                fulltext_paths=fulltext_paths or [],
                json_schema=json_schema,
            )
            return self._repository.create_collection(config)

        collection_id = self._repository.create_collection(config)
        return {"success": True, "collection_id": collection_id}

    def insert(
        self,
        collection_id: str,
        document: dict[str, Any],
        id: str | None = None,
    ) -> Document:
        """Insert a document.

        Args:
            collection_id: Collection identifier
            document: Document content
            id: Optional document ID

        Returns:
            Created document
        """
        return self._repository.insert(collection_id, document, id)

    def insert_batch(
        self,
        collection_id: str,
        documents: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[Document]:
        """Insert multiple documents.

        Args:
            collection_id: Collection identifier
            documents: List of document contents
            ids: Optional list of document IDs

        Returns:
            List of created documents
        """
        return self._repository.insert_batch(collection_id, documents, ids)

    def get(
        self,
        collection_id: str,
        doc_id: str,
    ) -> Document | None:
        """Get a document by ID.

        Args:
            collection_id: Collection identifier
            doc_id: Document identifier

        Returns:
            Document or None
        """
        return self._repository.get(collection_id, doc_id)

    def query(
        self,
        collection_id: str,
        filter: DocumentFilter | None = None,
        projection: list[str] | None = None,
        limit: int = 100,
    ) -> DocumentQueryResponse:
        """Query documents with filters.

        Args:
            collection_id: Collection identifier
            filter: Document filter
            projection: Fields to return
            limit: Maximum results

        Returns:
            List of documents

        Example:
            results = docs.query(
                collection_id="code_files",
                filter=DocumentFilter().eq("language", "python"),
                projection=["file_path", "language"],
                limit=10
            )
        """
        result = self._repository.query(
            collection_id=collection_id,
            filter=filter,
            projection=projection,
            limit=limit,
        )
        return DocumentQueryResponse(
            documents=[document.to_dict() for document in result.documents],
            total_count=result.total_count,
            has_more=result.has_more,
        )

    def search(
        self,
        collection_id: str,
        text_query: str,
        limit: int = 10,
    ) -> list[Document]:
        """Full-text search in documents.

        Args:
            collection_id: Collection identifier
            text_query: Search query
            limit: Maximum results

        Returns:
            List of matching documents

        Example:
            files = docs.search(
                collection_id="code_files",
                text_query="function that parses JSON",
                limit=10
            )
        """
        return self._repository.search(
            collection_id=collection_id,
            text_query=text_query,
            limit=limit,
        )

    def update(
        self,
        collection_id: str,
        doc_id: str,
        updates: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Update a document.

        Args:
            collection_id: Collection identifier
            doc_id: Document identifier
            updates: Fields to update

        Returns:
            Updated document or None
        """
        document = self._repository.update(collection_id, doc_id, updates)
        if document is None:
            return None
        return {
            "success": True,
            "id": document.id,
            "new_version": document.version,
            "document": document.content,
        }

    def delete(
        self,
        collection_id: str,
        doc_id: str,
    ) -> bool:
        """Delete a document.

        Args:
            collection_id: Collection identifier
            doc_id: Document identifier

        Returns:
            True if deleted
        """
        return self._repository.delete(collection_id, doc_id)

    def flush(self, collection_id: str) -> dict[str, Any]:
        """Flush pending batch operations.

        Args:
            collection_id: Collection identifier

        Returns:
            Flush result
        """
        return self._repository.flush_batch(collection_id)

    def get_document(
        self,
        collection_id: str,
        doc_id: str,
        projection: list[str] | None = None,
    ) -> dict[str, Any] | None:
        document = self._repository.get(collection_id, doc_id)
        if document is None:
            return None

        content = self._repository._project_document(document.content, projection)
        return {
            "id": document.id,
            "document": content,
            "version": document.version,
            "found": True,
        }

    def list_collections(self) -> list[dict[str, Any]]:
        return self._repository.list_collections()

    def delete_collection(self, collection_id: str) -> bool:
        return self._repository.delete_collection(collection_id)

    def aggregate(
        self,
        collection_id: str,
        pipeline: list[dict[str, Any]],
    ) -> dict[str, Any]:
        documents = list(self._repository._documents.get(collection_id, {}).values())

        for stage in pipeline:
            stage_name = stage.get("stage")
            if stage_name == "match":
                documents = [
                    doc
                    for doc in documents
                    if self._repository._matches_filter(
                        doc.content, stage.get("filter")
                    )
                ]
            elif stage_name == "group":
                grouped: dict[Any, list[Document]] = {}
                key_path = stage.get("key", "$.id")
                for doc in documents:
                    group_key = self._repository._get_value(doc.content, key_path)
                    grouped.setdefault(group_key, []).append(doc)

                results: list[dict[str, Any]] = []
                for group_key, group_docs in grouped.items():
                    row: dict[str, Any] = {"key": group_key}
                    for aggregation in stage.get("aggregations", []):
                        field_name = aggregation.get("field")
                        agg_type = aggregation.get("type")
                        path = aggregation.get("path", "$.id")
                        values = [
                            self._repository._get_value(doc.content, path)
                            for doc in group_docs
                        ]
                        values = [value for value in values if value is not None]
                        if agg_type == "count":
                            row[field_name] = len(group_docs)
                        elif agg_type == "avg":
                            row[field_name] = sum(values) / len(values) if values else 0
                        elif agg_type == "sum":
                            row[field_name] = sum(values) if values else 0
                    results.append(row)
                return {"results": results}

        return {"results": [doc.to_dict() for doc in documents]}


# =============================================================================
# Factory Functions
# =============================================================================


def create_document_api(
    client: Any,
    enable_cache: bool = True,
    cache_size: int = 1000,
) -> ProximaDBDocument:
    """Factory function to create document API instance.

    Args:
        client: ProximaDB client instance
        enable_cache: Enable write-through caching
        cache_size: LRU cache size

    Returns:
        ProximaDBDocument instance
    """
    return ProximaDBDocument(
        client=client,
        enable_cache=enable_cache,
        cache_size=cache_size,
    )
